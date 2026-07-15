#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/scan.h"

using hyle::simd::active_scan_variant;
using hyle::simd::best_isa;
using hyle::simd::exclusive_scan;
using hyle::simd::inclusive_scan;
using hyle::simd::Isa;

namespace {

template <typename T>
std::vector<T> iota_vals(std::size_t n, T start = T{1}) {
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(start + static_cast<T>(i));
  return v;
}

template <typename T>
std::vector<T> ref_inclusive(const std::vector<T>& in) {
  std::vector<T> o = in;
  for (std::size_t i = 1; i < o.size(); ++i) o[i] = static_cast<T>(o[i - 1] + o[i]);
  return o;
}

template <typename T>
std::vector<T> ref_exclusive(const std::vector<T>& in) {
  std::vector<T> o(in.size());
  T acc = T{0};
  for (std::size_t i = 0; i < in.size(); ++i) {
    o[i] = acc;
    acc = static_cast<T>(acc + in[i]);
  }
  return o;
}

template <typename T>
void check_inclusive(std::size_t n) {
  auto in = iota_vals<T>(n);
  auto expect = ref_inclusive(in);
  inclusive_scan(in.data(), n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(in[i], expect[i]) << "inc n=" << n << " i=" << i;
}

template <typename T>
void check_exclusive(std::size_t n) {
  auto in = iota_vals<T>(n);
  auto expect = ref_exclusive(in);
  exclusive_scan(in.data(), n);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(in[i], expect[i]) << "exc n=" << n << " i=" << i;
}

}  // namespace

TEST(Scan, InclusiveEmpty) {
  inclusive_scan(static_cast<std::uint32_t*>(nullptr), 0);
  SUCCEED();
}

TEST(Scan, ExclusiveEmpty) {
  exclusive_scan(static_cast<std::uint32_t*>(nullptr), 0);
  SUCCEED();
}

// Sizes hit tail, one vector, multi-vector, and odd tails for 128/256/512-bit widths.
TEST(Scan, InclusiveU32Sizes) {
  for (std::size_t n : {1u, 2u, 3u, 4u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 64u, 100u, 257u}) {
    check_inclusive<std::uint32_t>(n);
  }
}

TEST(Scan, ExclusiveU32Sizes) {
  for (std::size_t n : {1u, 2u, 3u, 4u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 64u, 100u, 257u}) {
    check_exclusive<std::uint32_t>(n);
  }
}

TEST(Scan, InclusiveI32) { check_inclusive<std::int32_t>(100); }
TEST(Scan, ExclusiveI32) { check_exclusive<std::int32_t>(100); }
TEST(Scan, InclusiveU64) { check_inclusive<std::uint64_t>(64); }
TEST(Scan, ExclusiveU64) { check_exclusive<std::uint64_t>(64); }
TEST(Scan, InclusiveI64) { check_inclusive<std::int64_t>(64); }
TEST(Scan, ExclusiveI64) { check_exclusive<std::int64_t>(64); }

TEST(Scan, InclusiveF32) {
  for (std::size_t n : {1u, 8u, 16u, 33u, 100u}) {
    auto in = iota_vals<float>(n, 0.5f);
    auto expect = ref_inclusive(in);
    inclusive_scan(in.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_FLOAT_EQ(in[i], expect[i]) << "f32 inc n=" << n << " i=" << i;
  }
}

TEST(Scan, ExclusiveF32) {
  for (std::size_t n : {1u, 8u, 16u, 33u, 100u}) {
    auto in = iota_vals<float>(n, 0.5f);
    auto expect = ref_exclusive(in);
    exclusive_scan(in.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_FLOAT_EQ(in[i], expect[i]) << "f32 exc n=" << n << " i=" << i;
  }
}

TEST(Scan, InclusiveF64) {
  auto in = iota_vals<double>(50, 1.0);
  auto expect = ref_inclusive(in);
  inclusive_scan(in.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) EXPECT_DOUBLE_EQ(in[i], expect[i]);
}

TEST(Scan, ExclusiveF64) {
  auto in = iota_vals<double>(50, 1.0);
  auto expect = ref_exclusive(in);
  exclusive_scan(in.data(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) EXPECT_DOUBLE_EQ(in[i], expect[i]);
}

TEST(Scan, ActiveVariantIsRealTier) {
  const Isa a = active_scan_variant();
#if defined(HYLE_ARCH_X64)
  EXPECT_GE(static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(Isa::kSse42));
  EXPECT_LE(static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(best_isa()));
#elif defined(HYLE_ARCH_ARM64)
  EXPECT_EQ(a, Isa::kNeon);
#else
  EXPECT_EQ(a, Isa::kScalar);
#endif
}
