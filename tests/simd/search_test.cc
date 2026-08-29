#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "hyle/simd/search.h"
#include "src/simd/runtime_internal.h"

namespace hs = hyle::simd;

namespace {

template <typename T>
T sample(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 23) - 11);
}

template <typename T>
void check_search_type() {
  for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{3},
                              std::size_t{7}, std::size_t{16}, std::size_t{31},
                              std::size_t{65}, std::size_t{257}}) {
    std::vector<T> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = sample<T>(i);
    const T needle = static_cast<T>(3);
    const auto ref_count = static_cast<std::size_t>(
        std::count(data.begin(), data.end(), needle));
    const auto it = std::find(data.begin(), data.end(), needle);
    const std::size_t ref_find =
        it == data.end() ? n : static_cast<std::size_t>(it - data.begin());
    EXPECT_EQ(hs::count_equal(data.data(), n, needle), ref_count);
    EXPECT_EQ(hs::find_equal(data.data(), n, needle), ref_find);

    const T missing = static_cast<T>(99);
    EXPECT_EQ(hs::find_equal(data.data(), n, missing), n);

    T got_min{}, got_max{};
    hs::minmax_value(data.data(), n, &got_min, &got_max);
    T min_identity;
    T max_identity;
    if constexpr (std::numeric_limits<T>::has_infinity) {
      min_identity = std::numeric_limits<T>::infinity();
      max_identity = -std::numeric_limits<T>::infinity();
    } else {
      min_identity = std::numeric_limits<T>::max();
      max_identity = std::numeric_limits<T>::lowest();
    }
    const T ref_min = n == 0 ? min_identity
                             : *std::min_element(data.begin(), data.end());
    const T ref_max = n == 0 ? max_identity
                             : *std::max_element(data.begin(), data.end());
    EXPECT_EQ(hs::min_value(data.data(), n), ref_min);
    EXPECT_EQ(hs::max_value(data.data(), n), ref_max);
    const std::size_t ref_min_index =
        n == 0 ? n : static_cast<std::size_t>(
                          std::min_element(data.begin(), data.end()) - data.begin());
    const std::size_t ref_max_index =
        n == 0 ? n : static_cast<std::size_t>(
                          std::max_element(data.begin(), data.end()) - data.begin());
    EXPECT_EQ(hs::index_of_min(data.data(), n), ref_min_index);
    EXPECT_EQ(hs::index_of_max(data.data(), n), ref_max_index);
    EXPECT_EQ(got_min, ref_min);
    EXPECT_EQ(got_max, ref_max);
  }
}

template <typename T>
void check_sorted_type() {
  for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                              std::size_t{7}, std::size_t{31}, std::size_t{65}}) {
    std::vector<T> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<T>(i / 2);
    EXPECT_TRUE(hs::is_sorted(data.data(), n, true));
    if (n >= 2) {
      data[n / 2] = std::numeric_limits<T>::lowest();
      EXPECT_EQ(hs::is_sorted(data.data(), n, true),
                std::is_sorted(data.begin(), data.end()));
    }

    for (std::size_t i = 0; i < n; ++i)
      data[i] = static_cast<T>((n - i) / 2);
    EXPECT_TRUE(hs::is_sorted(data.data(), n, false));
    if (n >= 2) {
      data[n / 2] = std::numeric_limits<T>::max();
      EXPECT_EQ(hs::is_sorted(data.data(), n, false),
                std::is_sorted(data.begin(), data.end(), std::greater<>{}));
    }
  }
}

}  // namespace

TEST(Search, AllIntegerTypes) {
  check_search_type<std::int8_t>();
  check_search_type<std::uint8_t>();
  check_search_type<std::int16_t>();
  check_search_type<std::uint16_t>();
  check_search_type<std::int32_t>();
  check_search_type<std::uint32_t>();
  check_search_type<std::int64_t>();
  check_search_type<std::uint64_t>();
}

TEST(Search, FloatingTypes) {
  check_search_type<float>();
  check_search_type<double>();
}

TEST(Search, SortedAllTypes) {
  check_sorted_type<std::int8_t>();
  check_sorted_type<std::uint8_t>();
  check_sorted_type<std::int16_t>();
  check_sorted_type<std::uint16_t>();
  check_sorted_type<std::int32_t>();
  check_sorted_type<std::uint32_t>();
  check_sorted_type<std::int64_t>();
  check_sorted_type<std::uint64_t>();
  check_sorted_type<float>();
  check_sorted_type<double>();
}

TEST(Search, FloatingNaNContract) {
  for (const bool use_double : {false, true}) {
    if (use_double) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      const double data[] = {1.0, nan, -4.0, 8.0};
      EXPECT_EQ(hs::count_equal(data, 4, nan), 0u);
      EXPECT_EQ(hs::find_equal(data, 4, nan), 4u);
      EXPECT_TRUE(std::isnan(hs::min_value(data, 4)));
      EXPECT_TRUE(std::isnan(hs::max_value(data, 4)));
      EXPECT_EQ(hs::index_of_min(data, 4), 1u);
      EXPECT_EQ(hs::index_of_max(data, 4), 1u);
      double lo{}, hi{};
      hs::minmax_value(data, 4, &lo, &hi);
      EXPECT_TRUE(std::isnan(lo));
      EXPECT_TRUE(std::isnan(hi));
      EXPECT_EQ(hs::is_sorted(data, 4), std::is_sorted(data, data + 4));
    } else {
      const float nan = std::numeric_limits<float>::quiet_NaN();
      const float data[] = {1.0f, nan, -4.0f, 8.0f};
      EXPECT_EQ(hs::count_equal(data, 4, nan), 0u);
      EXPECT_EQ(hs::find_equal(data, 4, nan), 4u);
      EXPECT_TRUE(std::isnan(hs::min_value(data, 4)));
      EXPECT_TRUE(std::isnan(hs::max_value(data, 4)));
      EXPECT_EQ(hs::index_of_min(data, 4), 1u);
      EXPECT_EQ(hs::index_of_max(data, 4), 1u);
    }
  }
}

TEST(Search, FloatingExtremaKeepFirstTieAndFirstNaNAcrossBlocks) {
  for (const hs::Runtime* runtime : {
           hs::runtime_for(hs::Isa::kScalar),
           hs::runtime_for(hs::Isa::kSse42),
           hs::runtime_for(hs::Isa::kAvx2),
           hs::runtime_for(hs::Isa::kAvx512),
           hs::runtime_for(hs::Isa::kNeon),
           hs::runtime_for(hs::Isa::kSve),
           hs::runtime_for(hs::Isa::kRvv)}) {
    if (runtime == nullptr) continue;
    std::vector<float> data(513, 4.0f);
    data[9] = -17.0f;
    data[301] = -17.0f;
    data[13] = 29.0f;
    data[477] = 29.0f;
    EXPECT_EQ(runtime->search.index_of_min_f32(data.data(), data.size()), 9u);
    EXPECT_EQ(runtime->search.index_of_max_f32(data.data(), data.size()), 13u);

    data[71] = std::numeric_limits<float>::quiet_NaN();
    data[402] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(runtime->search.index_of_min_f32(data.data(), data.size()), 71u);
    EXPECT_EQ(runtime->search.index_of_max_f32(data.data(), data.size()), 71u);
    EXPECT_TRUE(std::isnan(
        runtime->search.min_value_f32(data.data(), data.size())));
    EXPECT_TRUE(std::isnan(
        runtime->search.max_value_f32(data.data(), data.size())));
    float minimum = 0.0f;
    float maximum = 0.0f;
    runtime->search.minmax_value_f32(data.data(), data.size(), &minimum,
                                     &maximum);
    EXPECT_TRUE(std::isnan(minimum));
    EXPECT_TRUE(std::isnan(maximum));
  }
}

TEST(Search, EmptyNullRange) {
  EXPECT_EQ(hs::count_equal(static_cast<const std::uint32_t*>(nullptr), 0, 1), 0u);
  EXPECT_EQ(hs::find_equal(static_cast<const std::uint32_t*>(nullptr), 0, 1), 0u);
  EXPECT_EQ(hs::index_of_min(static_cast<const std::uint32_t*>(nullptr), 0), 0u);
  EXPECT_EQ(hs::index_of_max(static_cast<const std::uint32_t*>(nullptr), 0), 0u);
  EXPECT_TRUE(hs::is_sorted(static_cast<const std::uint32_t*>(nullptr), 0));
  EXPECT_EQ(hs::min_value(static_cast<const std::int32_t*>(nullptr), 0),
            std::numeric_limits<std::int32_t>::max());
  EXPECT_EQ(hs::max_value(static_cast<const std::int32_t*>(nullptr), 0),
            std::numeric_limits<std::int32_t>::lowest());
  EXPECT_EQ(hs::min_value(static_cast<const float*>(nullptr), 0),
            std::numeric_limits<float>::infinity());
  EXPECT_EQ(hs::max_value(static_cast<const float*>(nullptr), 0),
            -std::numeric_limits<float>::infinity());
  std::uint32_t minimum = 0;
  std::uint32_t maximum = 0;
  hs::minmax_value(static_cast<const std::uint32_t*>(nullptr), 0, &minimum,
                   &maximum);
  EXPECT_EQ(minimum, std::numeric_limits<std::uint32_t>::max());
  EXPECT_EQ(maximum, std::numeric_limits<std::uint32_t>::lowest());
}

TEST(Search, ExtremaIndexKeepsFirstTieAcrossCounterSegments) {
  std::vector<std::uint8_t> data(20000, 17);
  data[123] = 1;
  data[17000] = 1;
  data[456] = 251;
  data[19000] = 251;
  EXPECT_EQ(hs::index_of_min(data.data(), data.size()), 123u);
  EXPECT_EQ(hs::index_of_max(data.data(), data.size()), 456u);
}

TEST(Search, EverySupportedRuntimeVariant) {
  const std::int32_t data[] = {4, 1, 4, -3, 9, 4, 8};
  for (const hs::Isa isa : {hs::Isa::kScalar, hs::Isa::kSse42,
                            hs::Isa::kAvx2, hs::Isa::kAvx512,
                            hs::Isa::kNeon, hs::Isa::kSve, hs::Isa::kRvv}) {
    const hs::Runtime* rt = hs::runtime_for(isa);
    if (rt == nullptr) continue;
    EXPECT_EQ(rt->search.count_equal_i32(data, 7, 4), 3u);
    EXPECT_EQ(rt->search.find_equal_i32(data, 7, 9), 4u);
    EXPECT_EQ(rt->search.min_value_i32(data, 7), -3);
    EXPECT_EQ(rt->search.max_value_i32(data, 7), 9);
    EXPECT_EQ(rt->search.index_of_min_i32(data, 7), 3u);
    EXPECT_EQ(rt->search.index_of_max_i32(data, 7), 4u);
    std::int32_t lo{}, hi{};
    rt->search.minmax_value_i32(data, 7, &lo, &hi);
    EXPECT_EQ(lo, -3);
    EXPECT_EQ(hi, 9);
    EXPECT_FALSE(rt->search.is_sorted_asc_i32(data, 7));
    EXPECT_FALSE(rt->search.is_sorted_desc_i32(data, 7));
  }
}

TEST(Search, SelectedTargetDescribesActualKernelTable) {
  const auto& runtime = hs::runtime();
  const hs::SelectedTarget target =
      hs::selected_target(hs::KernelFamily::kSearch);
  EXPECT_EQ(target.isa, runtime.search_module.isa);
  EXPECT_EQ(target.features, runtime.search_module.features);
  EXPECT_NE(runtime.search.count_equal_i8, nullptr);
  EXPECT_NE(runtime.search.index_of_max_f64, nullptr);
}
