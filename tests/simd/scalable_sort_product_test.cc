#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "src/simd/runtime_internal.h"
#include "hyle/simd/sort.h"
#include "src/simd/sort/kernels.h"
#include "hwy/highway.h"

namespace hs = hyle::simd;
namespace hn = hwy::HWY_NAMESPACE;

#if defined(HYLE_SIMD_TARGET_RVV)
#define HYLE_SCALABLE_SORT(name) hs::detail::name##_rvv
#elif defined(HYLE_SIMD_TARGET_SVE)
#define HYLE_SCALABLE_SORT(name) hs::detail::name##_sve
#endif

namespace {

template <class T>
bool before(T a, T b, bool ascending) {
  if constexpr (std::is_floating_point_v<T>) {
    const bool an = std::isnan(a), bn = std::isnan(b);
    if (an || bn) return !an && bn;
  }
  return ascending ? a < b : a > b;
}

#define HYLE_DIRECT_ONE(T, tag)                                                        \
  void direct_sort(T* p, std::size_t n, bool asc) { HYLE_SCALABLE_SORT(sort_##tag)(p, n, asc); } \
  void direct_psort(T* p, std::size_t n, std::size_t k, bool asc) {                    \
    HYLE_SCALABLE_SORT(psort_##tag)(p, n, k, asc);                                     \
  }                                                                                    \
  void direct_select(T* p, std::size_t n, std::size_t k, bool asc) {                   \
    HYLE_SCALABLE_SORT(select_##tag)(p, n, k, asc);                                    \
  }
HYLE_DIRECT_ONE(std::uint32_t, u32)
HYLE_DIRECT_ONE(std::int32_t, i32)
HYLE_DIRECT_ONE(std::uint16_t, u16)
HYLE_DIRECT_ONE(std::int16_t, i16)
HYLE_DIRECT_ONE(std::uint64_t, u64)
HYLE_DIRECT_ONE(std::int64_t, i64)
HYLE_DIRECT_ONE(float, f32)
HYLE_DIRECT_ONE(double, f64)
#undef HYLE_DIRECT_ONE

template <class T>
std::vector<T> make_values(std::size_t n, unsigned pattern) {
  std::vector<T> out(n);
  std::uint64_t state = 0xD1B54A32D192ED03ULL + pattern;
  for (std::size_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    if (pattern == 1) out[i] = static_cast<T>(state & 7u);
    else if (pattern == 2) out[i] = static_cast<T>(i);
    else if (pattern == 3) out[i] = static_cast<T>(n - i);
    else if constexpr (std::is_floating_point_v<T>)
      out[i] = static_cast<T>(static_cast<std::int64_t>(state) * 0x1p-32);
    else
      out[i] = static_cast<T>(state);
  }
  return out;
}

template <class T>
void check_keys() {
  constexpr T kLeft = static_cast<T>(17);
  constexpr T kRight = static_cast<T>(29);
  for (const bool asc : {true, false}) {
    for (unsigned pattern = 0; pattern != 4; ++pattern) {
      constexpr std::size_t kSizes[] = {2, 3, 7, 16, 31, 64, 127, 257, 1000};
      for (std::size_t n : kSizes) {
        const auto input = make_values<T>(n, pattern);
        std::vector<T> storage(n + 16, kLeft);
        std::fill(storage.end() - 8, storage.end(), kRight);
        T* const got = storage.data() + 8;
        std::copy(input.begin(), input.end(), got);
        auto expected = input;
        std::sort(expected.begin(), expected.end(), [asc](T a, T b) { return before(a, b, asc); });
        direct_sort(got, n, asc);
        ASSERT_TRUE(std::equal(got, got + n, expected.begin()))
            << "sort n=" << n << " pattern=" << pattern << " asc=" << asc;
        EXPECT_TRUE(std::all_of(storage.begin(), storage.begin() + 8,
                                [](T x) { return x == kLeft; }));
        EXPECT_TRUE(std::all_of(storage.end() - 8, storage.end(),
                                [](T x) { return x == kRight; }));

        for (const std::size_t k : {std::size_t{1}, n / 2, n - 1}) {
          std::copy(input.begin(), input.end(), got);
          direct_psort(got, n, k, asc);
          ASSERT_TRUE(std::equal(got, got + k, expected.begin())) << "psort n=" << n << " k=" << k;

          std::copy(input.begin(), input.end(), got);
          direct_select(got, n, k, asc);
          ASSERT_EQ(got[k], expected[k]) << "select n=" << n << " k=" << k;
          for (std::size_t i = 0; i < k; ++i) EXPECT_FALSE(before(got[k], got[i], asc));
          for (std::size_t i = k + 1; i < n; ++i) EXPECT_FALSE(before(got[i], got[k], asc));
        }
      }
    }
  }
}

template <class T>
void check_nan() {
  const T nan = std::numeric_limits<T>::quiet_NaN();
  std::vector<T> values(257);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = i % 11 == 0 ? nan : static_cast<T>(static_cast<int>(i % 37) - 18);
  for (bool asc : {true, false}) {
    auto got = values;
    direct_sort(got.data(), got.size(), asc);
    const auto first_nan = std::find_if(got.begin(), got.end(), [](T x) { return std::isnan(x); });
    ASSERT_EQ(static_cast<std::size_t>(got.end() - first_nan), 24u);
    ASSERT_TRUE(std::is_sorted(got.begin(), first_nan,
                               [asc](T a, T b) { return before(a, b, asc); }));
  }
}

template <class KV, class K>
void check_kv(void (*sort_fn)(KV*, std::size_t, bool)) {
  for (bool asc : {true, false}) {
    std::vector<KV> values(257);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i].key = static_cast<K>((i * 37 + i / 5) & 31u);
      values[i].value = static_cast<K>(i + 1);
    }
    const auto original = values;
    sort_fn(values.data(), values.size(), asc);
    for (std::size_t i = 1; i < values.size(); ++i)
      ASSERT_FALSE(asc ? values[i].key < values[i - 1].key
                       : values[i].key > values[i - 1].key);
    auto by_value = [](const KV& a, const KV& b) { return a.value < b.value; };
    auto got_pairs = values;
    auto ref_pairs = original;
    std::sort(got_pairs.begin(), got_pairs.end(), by_value);
    std::sort(ref_pairs.begin(), ref_pairs.end(), by_value);
    for (std::size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(got_pairs[i].key, ref_pairs[i].key);
      ASSERT_EQ(got_pairs[i].value, ref_pairs[i].value);
    }
  }
}

}  // namespace

TEST(ScalableSortProduct, U32EntryPoints) { check_keys<std::uint32_t>(); }
TEST(ScalableSortProduct, I32EntryPoints) { check_keys<std::int32_t>(); }
TEST(ScalableSortProduct, U16EntryPoints) { check_keys<std::uint16_t>(); }
TEST(ScalableSortProduct, I16EntryPoints) { check_keys<std::int16_t>(); }
TEST(ScalableSortProduct, U64EntryPoints) { check_keys<std::uint64_t>(); }
TEST(ScalableSortProduct, I64EntryPoints) { check_keys<std::int64_t>(); }
TEST(ScalableSortProduct, F32EntryPoints) { check_keys<float>(); }
TEST(ScalableSortProduct, F64EntryPoints) { check_keys<double>(); }

TEST(ScalableSortProduct, FloatingNaNsRemainLast) {
  check_nan<float>(); check_nan<double>();
}

TEST(ScalableSortProduct, KeyValuePayloadsRemainAttached) {
  check_kv<hs::detail::KV32Bits, std::uint32_t>(
      [](auto* p, std::size_t n, bool asc) { HYLE_SCALABLE_SORT(sort_kv32)(p, n, asc); });
  check_kv<hs::detail::KV64Bits, std::uint64_t>(
      [](auto* p, std::size_t n, bool asc) { HYLE_SCALABLE_SORT(sort_kv64)(p, n, asc); });
}

TEST(ScalableSortProduct, U128EntryPoints) {
  using U = hs::detail::U128Bits;
  auto less = [](const U& a, const U& b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
  };
  std::vector<U> values(257);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = U{static_cast<std::uint64_t>(i * 0x9E3779B9u),
                  static_cast<std::uint64_t>((i * 37 + i / 5) & 31u)};
  auto expected = values;
  std::sort(expected.begin(), expected.end(), less);
  HYLE_SCALABLE_SORT(sort_u128)(values.data(), values.size(), true);
  for (std::size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(values[i].hi, expected[i].hi);
    ASSERT_EQ(values[i].lo, expected[i].lo);
  }
}

TEST(ScalableSortProduct, RuntimeAndPublicApiUseScalableTier) {
#if defined(HYLE_SIMD_TARGET_RVV)
  constexpr hs::KernelTarget kExpected = hs::KernelTarget::kRvv;
#else
  constexpr hs::KernelTarget kExpected = hs::KernelTarget::kSve;
#endif
  EXPECT_EQ(hs::runtime().sort.target, kExpected);
  const hs::Runtime* exact = hs::runtime_for(kExpected);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(exact->sort.target, kExpected);

  auto values = make_values<std::uint32_t>(1000, 0);
  hs::sort(values.data(), values.size(), hs::SortOrder::kAscending);
  EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));

  std::vector<hs::KeyValue32> kv(257);
  for (std::size_t i = 0; i < kv.size(); ++i) {
    kv[i].key = static_cast<std::uint32_t>((i * 29) & 31u);
    kv[i].value = static_cast<std::uint32_t>(i + 1);
  }
  hs::sort(kv.data(), kv.size(), hs::SortOrder::kDescending);
  for (std::size_t i = 1; i < kv.size(); ++i) EXPECT_GE(kv[i - 1].key, kv[i].key);
}

#if defined(HYLE_SIMD_TARGET_SVE)
TEST(ScalableSortProduct, SvePartitionAcceptsEveryPhysicalLaneCount) {
  const std::size_t full_lanes = hn::Lanes(hn::ScalableTag<std::uint32_t>{});
  const std::size_t n = 8 * full_lanes + 17;
  auto values = make_values<std::uint32_t>(n, 0);
  direct_sort(values.data(), values.size(), true);
  EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));
}
#endif

#undef HYLE_SCALABLE_SORT
