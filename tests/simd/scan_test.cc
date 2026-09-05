#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "hyle/simd/target.h"
#include "hyle/simd/scan.h"
#include "src/simd/runtime_internal.h"

namespace hs = hyle::simd;

namespace {

template <typename T>
T add_contract(T a, T b) noexcept {
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    return std::bit_cast<T>(
        static_cast<U>(std::bit_cast<U>(a) + std::bit_cast<U>(b)));
  } else {
    return static_cast<T>(a + b);
  }
}

template <typename T>
std::vector<T> values(std::size_t n) {
  std::vector<T> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      result[i] = static_cast<T>(static_cast<int>(i % 11) - 5) * T{0.25};
    } else if constexpr (std::is_signed_v<T>) {
      constexpr T pattern[] = {std::numeric_limits<T>::max(), T{1},
                               std::numeric_limits<T>::min(), T{-1}, T{7},
                               T{-9}};
      result[i] = pattern[i % (sizeof(pattern) / sizeof(pattern[0]))];
    } else {
      constexpr T pattern[] = {std::numeric_limits<T>::max(), T{1}, T{0},
                               T{7}, T{19}, T{31}};
      result[i] = pattern[i % (sizeof(pattern) / sizeof(pattern[0]))];
    }
  }
  return result;
}

template <typename T>
std::vector<T> reference_inclusive(const std::vector<T>& input) {
  std::vector<T> result(input.size());
  T accumulator{};
  for (std::size_t i = 0; i < input.size(); ++i) {
    accumulator = add_contract(accumulator, input[i]);
    result[i] = accumulator;
  }
  return result;
}

template <typename T>
std::vector<T> reference_exclusive(const std::vector<T>& input, T init) {
  std::vector<T> result(input.size());
  T accumulator = init;
  for (std::size_t i = 0; i < input.size(); ++i) {
    result[i] = accumulator;
    accumulator = add_contract(accumulator, input[i]);
  }
  return result;
}

template <typename T>
void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected,
                  const char* operation, std::size_t n) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_EQ(actual[i], expected[i])
          << operation << " n=" << n << " i=" << i;
    } else {
      EXPECT_EQ(actual[i], expected[i])
          << operation << " n=" << n << " i=" << i;
    }
  }
}

template <typename T, typename Inclusive, typename Exclusive>
void check_scan_shape(Inclusive inclusive, Exclusive exclusive) {
  constexpr T init = T{7};
  for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                              std::size_t{3}, std::size_t{7}, std::size_t{8},
                              std::size_t{15}, std::size_t{16},
                              std::size_t{17}, std::size_t{31},
                              std::size_t{32}, std::size_t{33},
                              std::size_t{65}, std::size_t{257}}) {
    const auto input = values<T>(n);
    const auto expected_inclusive = reference_inclusive(input);
    const auto expected_exclusive = reference_exclusive(input, init);

    std::vector<T> out(n);
    inclusive(input.data(), out.data(), n);
    expect_equal(out, expected_inclusive, "inclusive/out-of-place", n);

    auto in_place = input;
    inclusive(in_place.data(), in_place.data(), n);
    expect_equal(in_place, expected_inclusive, "inclusive/in-place", n);

    exclusive(input.data(), out.data(), n, init);
    expect_equal(out, expected_exclusive, "exclusive/out-of-place", n);

    in_place = input;
    exclusive(in_place.data(), in_place.data(), n, init);
    expect_equal(in_place, expected_exclusive, "exclusive/in-place", n);
  }
}

template <typename T>
void check_public_shape() {
  check_scan_shape<T>(
      [](const T* src, T* dst, std::size_t n) {
        hs::inclusive_scan(src, dst, n);
      },
      [](const T* src, T* dst, std::size_t n, T init) {
        hs::exclusive_scan(src, dst, n, init);
      });
}

}  // namespace

TEST(Scan, EmptyRangeAllowsNullPointers) {
  hs::inclusive_scan(static_cast<const std::uint32_t*>(nullptr),
                     static_cast<std::uint32_t*>(nullptr), 0);
  hs::exclusive_scan(static_cast<const std::uint32_t*>(nullptr),
                     static_cast<std::uint32_t*>(nullptr), 0, 17u);
  hs::inclusive_scan(static_cast<std::uint32_t*>(nullptr), 0);
  hs::exclusive_scan(static_cast<std::uint32_t*>(nullptr), 0);
}

TEST(Scan, PublicCoreAndConveniencesCoverAllTypes) {
  check_public_shape<std::uint32_t>();
  check_public_shape<std::int32_t>();
  check_public_shape<std::uint64_t>();
  check_public_shape<std::int64_t>();
  check_public_shape<float>();
  check_public_shape<double>();

  auto input = values<std::int32_t>(67);
  const auto expected_inclusive = reference_inclusive(input);
  hs::inclusive_scan(input.data(), input.size());
  expect_equal(input, expected_inclusive, "inclusive/convenience", input.size());

  input = values<std::int32_t>(67);
  const auto expected_zero = reference_exclusive(input, std::int32_t{});
  hs::exclusive_scan(input.data(), input.size());
  expect_equal(input, expected_zero, "exclusive/zero-convenience", input.size());
}

TEST(Scan, EverySupportedRuntimeVariant) {
  for (const hs::KernelTarget isa : {hs::KernelTarget::kScalar, hs::KernelTarget::kSse4,
                            hs::KernelTarget::kAvx2, hs::KernelTarget::kAvx512,
                            hs::KernelTarget::kNeon, hs::KernelTarget::kSve, hs::KernelTarget::kRvv}) {
    const hs::Runtime* runtime = hs::runtime_for(isa);
    if (runtime == nullptr) continue;

#define HYLE_CHECK_SCAN_RUNTIME(T, tag)                                      \
  check_scan_shape<T>(                                                       \
      [runtime](const T* src, T* dst, std::size_t n) {                       \
        runtime->scan_ops.inclusive_scan_##tag(src, dst, n);                 \
      },                                                                     \
      [runtime](const T* src, T* dst, std::size_t n, T init) {               \
        runtime->scan_ops.exclusive_scan_##tag(src, dst, n, init);           \
      })

    HYLE_CHECK_SCAN_RUNTIME(std::uint32_t, u32);
    HYLE_CHECK_SCAN_RUNTIME(std::int32_t, i32);
    HYLE_CHECK_SCAN_RUNTIME(std::uint64_t, u64);
    HYLE_CHECK_SCAN_RUNTIME(std::int64_t, i64);
    HYLE_CHECK_SCAN_RUNTIME(float, f32);
    HYLE_CHECK_SCAN_RUNTIME(double, f64);

#undef HYLE_CHECK_SCAN_RUNTIME
  }
}

TEST(Scan, SelectedTargetDescribesActualKernelTable) {
  const auto& runtime = hs::runtime();
  const hs::SelectedTarget target = hs::selected_target(hs::KernelFamily::kScan);
  EXPECT_EQ(target.target, runtime.scan.target);
  EXPECT_EQ(target.features, runtime.scan.features);
  EXPECT_NE(runtime.scan_ops.inclusive_scan_u32, nullptr);
  EXPECT_NE(runtime.scan_ops.exclusive_scan_f64, nullptr);
}
