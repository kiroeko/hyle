#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// Instantiate the actual vqsort body in this ISA test executable under a private suffix. This
// bypasses the public runtime ladder and proves the selected fixed backend itself, including on
// SDE/QEMU hosts where that backend is not the build machine's native runtime tier.
#define HYLE_SORT_TIER test
#include "src/simd/sort/vqsort_body.inc"
#undef HYLE_SORT_TIER

namespace hs = hyle::simd;

namespace {

template <class T>
void check_scalar_key(void (*sort_fn)(T*, std::size_t, bool),
                      void (*partial_fn)(T*, std::size_t, std::size_t, bool),
                      void (*select_fn)(T*, std::size_t, std::size_t, bool)) {
  std::mt19937_64 rng(0x501716ULL + sizeof(T));
  for (const std::size_t n : {std::size_t{17}, std::size_t{31}, std::size_t{64},
                              std::size_t{257}, std::size_t{1009}}) {
    std::vector<T> source(n);
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = static_cast<T>(i % 11 == 0 ? 7 : rng());
    for (bool ascending : {true, false}) {
      const auto before = [ascending](T a, T b) { return ascending ? a < b : a > b; };
      auto expected = source;
      std::sort(expected.begin(), expected.end(), before);

      auto got = source;
      sort_fn(got.data(), got.size(), ascending);
      for (std::size_t i = 0; i < n; ++i)
        ASSERT_EQ(got[i], expected[i]) << "sort n=" << n << " asc=" << ascending << " i=" << i;

      const std::size_t k = n / 3;
      got = source;
      partial_fn(got.data(), got.size(), k, ascending);
      for (std::size_t i = 0; i < k; ++i)
        ASSERT_EQ(got[i], expected[i]) << "partial n=" << n << " asc=" << ascending << " i=" << i;

      got = source;
      select_fn(got.data(), got.size(), k, ascending);
      ASSERT_EQ(got[k], expected[k]) << "select n=" << n << " asc=" << ascending;
      for (std::size_t i = 0; i < k; ++i) ASSERT_FALSE(before(got[k], got[i]));
      for (std::size_t i = k + 1; i < got.size(); ++i) ASSERT_FALSE(before(got[i], got[k]));
    }
  }
}

}  // namespace

TEST(SortKernel, U16DirectFixedTier) {
  check_scalar_key<std::uint16_t>(hs::detail::sort_u16_test, hs::detail::psort_u16_test,
                                  hs::detail::select_u16_test);
}

TEST(SortKernel, I16DirectFixedTier) {
  check_scalar_key<std::int16_t>(hs::detail::sort_i16_test, hs::detail::psort_i16_test,
                                 hs::detail::select_i16_test);
}

TEST(SortKernel, U128DirectFixedTier) {
  using U = hs::detail::U128Bits;
  const auto before = [](const U& a, const U& b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
  };
  std::mt19937_64 rng(0x1285017ULL);
  std::vector<U> source(1009);
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = U{i % 13 == 0 ? 0 : rng(), i % 7 == 0 ? 3 : rng()};
  auto expected = source;
  std::sort(expected.begin(), expected.end(), before);

  auto got = source;
  hs::detail::sort_u128_test(got.data(), got.size(), true);
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].hi, expected[i].hi);
    EXPECT_EQ(got[i].lo, expected[i].lo);
  }

  constexpr std::size_t k = 137;
  got = source;
  hs::detail::psort_u128_test(got.data(), got.size(), k, true);
  for (std::size_t i = 0; i < k; ++i) {
    EXPECT_EQ(got[i].hi, expected[i].hi);
    EXPECT_EQ(got[i].lo, expected[i].lo);
  }

  got = source;
  hs::detail::select_u128_test(got.data(), got.size(), k, true);
  EXPECT_EQ(got[k].hi, expected[k].hi);
  EXPECT_EQ(got[k].lo, expected[k].lo);
}
